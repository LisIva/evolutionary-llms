from evaluator import piped_evaluator

if __name__=='__main__':
    # 1. написать обработку нулевой итерации

    score, string_form_of_the_equation, params = piped_evaluator() # оценивает одну итерацию, evaluator не может предложить нулевую!

    # 2. написать ф-ю для переписывания промпта - там надо обновить exp_buffer с полученным score
    print()
