import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T


def _get_cart_location(env, screen_width):
    # x_threshold: maximum range of the cart to each side (in terms of units)
    world_width = env.x_threshold * 2
    # Finding the scale of the world relative to the screen
    scale = screen_width / world_width
    # Finding the x-axis location of the center of the cart on the screen
    # by mapping its per-unit location from its current state to the screen
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]

    # Now in order to crop the screen horizontally into a smaller screen of width='view_width',
    # We check the location of the cart center;
    # If it has enough margin from both edges of the original screen width,
    # we crop a rectangle with 'view_width/2' from each side of the cart center
    # (option III in the following condition).
    # However if the cart center is closer than 'view_width/2' to the screen edges,
    # then we select one of the first two options.
    view_width = int(screen_width * 0.6)
    cart_location = _get_cart_location(env, screen_width)
    if cart_location < (view_width // 2):
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)

    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]

    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

    # Resize, and add a batch dimension (BCHW)
    final_torch_screen = resize(screen).cpu().numpy()
    return final_torch_screen
