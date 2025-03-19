from typing import Union

import pygame
from pygame import Vector2

import numpy as np


class LRPoint:
    def __init__(self, position: Vector2, radius: float):
        self.position = position
        self.radius = radius

    def update(self, new_position: Vector2):
        self.position = new_position

    def clicked(self, cursor_pos: Vector2):
        return (cursor_pos - self.position).magnitude() <= self.radius

    def draw(self, surface: pygame.Surface, color: pygame.Color):
        pygame.draw.circle(surface, color, self.position, self.radius)


class LRInteractive(pygame.Surface):
    def __init__(self, width, height, degree=1):
        super().__init__((width, height))
        self.points: list[LRPoint] = []
        self.degree = degree

        #self._gradient = None
        #self._intercept = None
        self._coeffs = None

    # @property
    # def gradient(self):
    #     return self._gradient
    #
    # @property
    # def intercept(self):
    #     return self._intercept

    @property
    def coeffs(self):
        return self._coeffs

    def place_point(self, point: Vector2, radius: float):
        if self.get_rect().collidepoint(point.x, point.y):
            self.points.append(LRPoint(point, radius))
            self.update_line_params()

    def remove_point_at_click(self, click_pos):
        for i in range(len(self.points) - 1, -1, -1):
            point = self.points[i]
            if point.clicked(click_pos):
                self.points.pop(i)
                self.update_line_params()
                break

    def update_line_params(self):
        if len(self.points) < 2:
            return

        xs = []
        ys = []
        for point in self.points:
            xs.append(point.position.x)
            ys.append(point.position.y)

        xs = np.array(xs)
        ys = np.array(ys)

        A = np.tile(xs.reshape(-1, 1), (1, self.degree + 1)) ** np.arange(self.degree + 1)
        coeffs, loss, _, _ = np.linalg.lstsq(A, ys)
        #n = len(xs)

        self._coeffs = coeffs
        #self._gradient = (n*np.sum(xs * ys) - np.sum(xs) * np.sum(ys)) / (n * np.sum(xs ** 2) - np.sum(xs) ** 2)
        #self._intercept = (np.sum(ys) * np.sum(xs ** 2) - np.sum(xs) * np.sum(xs * ys)) / (n * np.sum(xs ** 2) - np.sum(xs) ** 2)

    def predict(self, xs):
        if isinstance(xs, int):
            xs = np.array([xs])

        #return self._gradient * x + self._intercept
        left = (np.tile(xs.reshape(-1,1), (1,self.degree + 1)) ** np.arange(self.degree + 1))
        right = self._coeffs.reshape(-1,1)
        return left @ right

    def get_point_maybe(self, pos: Vector2) -> Union[None, LRPoint]:
        for i in range(len(self.points) - 1, -1, -1):
            point = self.points[i]
            if point.clicked(pos):
                return point

    def draw(self):
        self.fill("white")
        pygame.draw.rect(self,
                         "black",
                         self.get_rect(),
                         1)

        for point in self.points:
            point.draw(self, pygame.Color(0, 100, 0))

        #if self._gradient is not None and self._intercept is not None:
        if self._coeffs is not None:
            sample_xs = np.linspace(0, self.get_width(), 2000)
            sample_ys = self.predict(sample_xs)
            #start = Vector2(0, self.predict(0))
            #end = Vector2(self.get_width(), self.predict(self.get_width()))
            points: list[Vector2] = [Vector2(x,y) for x,y in zip(sample_xs, sample_ys)]

            pygame.draw.lines(self, "blue", False, points)

def main():
    pygame.init()
    clock = pygame.time.Clock()
    fps = 180

    background_color = "white"

    screen_width, screen_height = 1366, 768
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pygame Linear Regression")

    interactive = LRInteractive(screen_width * 4/5, screen_height, 3)
    interactive_top_left = Vector2(screen_width / 5, 0)

    grabbed_point = None
    running = True
    lshift_holding = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LSHIFT:
                    lshift_holding = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LSHIFT:
                    lshift_holding = False
                    grabbed_point = None

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and lshift_holding:
                    grabbed_point = interactive.get_point_maybe(Vector2(*pygame.mouse.get_pos()) - interactive_top_left)
                    print(grabbed_point)
                elif event.button == 1:
                    interactive.place_point(Vector2(*pygame.mouse.get_pos()) - interactive_top_left, 5)
                elif event.button == 3:
                    interactive.remove_point_at_click(Vector2(*pygame.mouse.get_pos()) - interactive_top_left)


        if grabbed_point is not None:
            grabbed_point.update(Vector2(*pygame.mouse.get_pos()) - interactive_top_left)
            interactive.update_line_params()

        screen.fill(background_color)
        interactive.draw()
        screen.blit(interactive, interactive_top_left)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()

if __name__ == "__main__":
    main()
