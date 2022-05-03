# BASED ON https://api.arcade.academy/en/latest/examples/array_backed_grid.html

import pickle
import arcade
import math
import time

# actual execution speed is ~0.05
TICK_SPEED = 0.4

ROW_COUNT = 20
COLUMN_COUNT = 10

WIDTH = 30
HEIGHT = 30

MARGIN = 5

SCREEN_WIDTH = (WIDTH + MARGIN) * COLUMN_COUNT + MARGIN
SCREEN_HEIGHT = (HEIGHT + MARGIN) * ROW_COUNT + MARGIN
SCREEN_TITLE = "NEATris"


class Tetris(arcade.Window):
    def __init__(self, width, height, title, log_to_render):
        super().__init__(width, height, title)
        self.log = log_to_render
        self.dt = 0
        self.start_delay = 5

        arcade.set_background_color(arcade.color.BLACK)

    def on_draw(self):
        self.clear()

        # Draw the grid
        current_frame_number = math.floor(self.dt / TICK_SPEED)
        current_frame = self.log[min(current_frame_number, len(self.log) - 1)]
        for row in range(ROW_COUNT):
            for column in range(COLUMN_COUNT):
                # Figure out what color to draw the box
                if current_frame['board'][column] > row:
                    color = arcade.color.CYAN
                else:
                    color = arcade.color.BLACK

                # Do the math to figure out where the box is
                x = (MARGIN + WIDTH) * column + MARGIN + WIDTH // 2
                y = (MARGIN + HEIGHT) * row + MARGIN + HEIGHT // 2

                # Draw the box
                arcade.draw_rectangle_filled(x, y, WIDTH, HEIGHT, color)

        arcade.draw_text(f"Rows Cleared: {current_frame['score']}",
                         0,
                         SCREEN_HEIGHT-40,
                         arcade.color.WHITE,
                         25,
                         width=SCREEN_WIDTH,
                         align="center")

    def on_update(self, delta_time: float):
        if self.start_delay > 0:
            self.start_delay -= delta_time
        else:
            self.dt += delta_time



def main():
    with open('log.pickle', 'rb') as handle:
        game_log = pickle.load(handle)
    Tetris(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, log_to_render=game_log)
    arcade.run()


if __name__ == "__main__":
    main()
