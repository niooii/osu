import pygame

from ..rulesets import hitobjects as hitobjects

def _render_hitcircle(hitcircle, time: int, screen: pygame.Surface, preempt: float, fade_in: float, color:pygame.Color, circle_radius:int, *args):
	surface = pygame.Surface((circle_radius * 2, circle_radius * 2))
	surface.set_colorkey((0, 0, 0))
	pygame.draw.circle(surface, color, (circle_radius, circle_radius), circle_radius)

	alpha = min([1, (time - hitcircle.time + preempt) / fade_in])
	surface.set_alpha(alpha * 127)
	
	pos = (hitcircle.x - circle_radius, hitcircle.y - circle_radius)
	screen.blit(surface, pos)


def _render_slider(slider, time: int, screen: pygame.Surface, preempt: float, fade_in: float,  color:pygame.Color, circle_radius:int, beat_duration:float, multiplier:float=1.0):
	curve_points = slider._get_curve_points()
	
	if len(curve_points) > 1:
		pygame.draw.lines(screen, (255, 255, 255), False, curve_points, 2)

	vertices = [(slider.x, slider.y)] + slider.curve_points
	
	if len(vertices) > 1:
		pygame.draw.lines(screen, (128, 128, 128), False, vertices, 1)
	
	vertex_size = 4
	for vertex in vertices:
		x, y = vertex
		rect = pygame.Rect(x - vertex_size//2, y - vertex_size//2, vertex_size, vertex_size)
		pygame.draw.rect(screen, (128, 128, 128), rect)

	#  slider type text
	slider_type_text = slider.slider_type.value  # (B, L, P, C)
	font = pygame.font.Font(None, 25)
	text_surface = font.render(slider_type_text, True, (200, 200, 200))
	
	text_x = slider.x + 25
	text_y = slider.y - 25
	screen.blit(text_surface, (text_x, text_y))

	_render_hitcircle(slider,  time, screen, preempt, fade_in, color, circle_radius)
	
	pos = slider.target_position(time, beat_duration, multiplier)
	pygame.draw.circle(screen, (255, 255, 255), list(map(int, pos)), circle_radius, 1)


SPINNER_RADIUS = 128


def _render_spinner(spinner, time: int, screen: pygame.Surface,  *args):
	pos = (spinner.x, spinner.y)
	pygame.draw.circle(screen, (255, 255, 255), pos, SPINNER_RADIUS, 2)


def render(obj: hitobjects.HitObject, time: int, screen: pygame.Surface, *args):
	if isinstance(obj, hitobjects.HitCircle):
		_render_hitcircle(obj, time, screen, *args)
	if isinstance(obj, hitobjects.Slider):
		_render_slider(obj, time, screen, *args)
	if isinstance(obj, hitobjects.Spinner):
		_render_spinner(obj, time, screen, *args)