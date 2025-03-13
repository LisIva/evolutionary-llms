import re
from promptconstructor.info_prompts import prompt_complete_inf
# from pipeline.buffer_handler.sub_eq_builder import one_stroke_rs_code, strip, split_with_braces


class Individ(object):
    def __init__(self, key, llm_form, epde_form):
        pass


# 1. Вместо c[..] подставить реальные коэффы из record_track
# 2. Написать еще {power: 1} и разобрать отдельно кейсы с **
# 3. Можно написать класс детектор и класс клинер и класс преобразователя
# 4. Detector также возвращает необычные элементы (t, x, sin(t), ...)
class Translator(object):
    def __init__(self, record_track: dict, populat_track: dict):
        self.record_track = record_track
        self.populat_track = populat_track

    def some(self):
        for sol_key in self.populat_track.keys():
            # sol_terms = strip(split_with_braces(sol_key))
            pass
            # вместо c[..] подставить реальные коэффы из record_track


if __name__ == '__main__':
    pop_track = {'du/dt = c[0] * du/dx + c[1] * t * du/dx + c[2] * t * x': (1.6, 460.5686610664196), 'du/dt = c[0] * du/dx + c[1] * u + c[2] * d^2u/dx^2': (1.45, 484.1114426561667), 'du/dt = c[0] * du/dx + c[1] * u * du/dx': (1.2, 438.94292729549943), 'du/dt = c[0] * u * du/dx + c[1] * du/dx + c[2] * d^2u/dx^2': (1.95, 37.14800565887713), 'du/dt = c[0] * u * du/dx + c[1] * d^2u/dx^2': (1.45, 38.90635312678824), 'du/dt = c[0] * u * du/dx + c[1] * d^2u/dx^2 + c[2] * du/dx * t': (2.15, 37.057907826954576), 'du/dt = c[0] * du/dx + c[1] * du/dt * d^2u/dx^2': (1.75, 542.9853705131861), 'du/dt = c[0] * du/dx + c[1] * t * du/dx': (1.2, 442.49077370655203)}
    trans = Translator()
    print()