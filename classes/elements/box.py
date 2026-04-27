import matplotlib.patches as patches

class Box:
    def __init__(self, box_id, x, y, w=0.4, h=0.4, color='red', obstacle_box=False):
        self.id           = box_id
        self.x            = x
        self.y            = y
        self.w            = w
        self.h            = h
        self.color        = color
        self.obstacle_box = obstacle_box
        self.stacked      = False
        self.stack_layer  = -1
        self.carried_by   = None
        self.patch = None
        self.text_artist = None

    def setup_visuals(self, ax):
        """Creates the visual elements and adds them to the axes."""
        self.patch = patches.Rectangle(
            (self.x - self.w/2, self.y - self.h/2), self.w, self.h,
            fc=self.color, ec='white', lw=1.2, zorder=4
        )
        ax.add_patch(self.patch)
        self.text_artist = ax.text(
            self.x, self.y, self.id, color='white',
            ha='center', va='center', fontsize=8,
            fontfamily='monospace', fontweight='bold', zorder=5
        )
        return [self.patch, self.text_artist]

    def update_visuals(self):
        """Updates visual state per frame."""
        self.patch.set_xy((self.x - self.w/2, self.y - self.h/2))
        if self.stacked:
            self.patch.set_alpha(0.6)
        self.text_artist.set_position((self.x, self.y))
        
    def get_bounds(self):
        """Returns (x_min, x_max, y_min, y_max) based on width and height."""
        return (self.x - self.w/2, self.x + self.w/2, 
                self.y - self.h/2, self.y + self.h/2)