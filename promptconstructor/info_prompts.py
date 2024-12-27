# дописать случаи когда на левой стороне не та производная

prompt_complete_inf = {
    'burg': {'full_form': "du/dt = F(t, x, u, du/dx)",
             'dots_order': "t x u du/dt du/dx",
             'left_deriv': 'du/dt',},

    'sindy-burg': {'full_form': "du/dt = F(t, x, u, du/dx, d^2u/dt^2, d^2u/dx^2)",
                   'dots_order': "t x u du/dt du/dx d^2u/dt^2 d^2u/dx^2",
                   'left_deriv': 'du/dt',},

    'kdv': {'full_form': "du/dt = F(t, x, u, du/dx, d^2u/dx^2, d^3u/dx^3)",
            'dots_order': 't x u du/dt du/dx d^2u/dx^2 d^3u/dx^3',
            'left_deriv': 'du/dt', },

    'sindy-kdv': {'full_form': "du/dt = F(t, x, u, du/dx, d^2u/dt^2, d^2u/dx^2, d^3u/dx^3)",
                  'dots_order': 't x u du/dt du/dx d^2u/dt^2 d^2u/dx^2 d^3u/dx^3',
                  'left_deriv': 'du/dt', },

    'wave': {'full_form': "d^2u/dt^2 = F(t, x, u, du/dt, du/dx, d^2u/dx^2)",
             'dots_order': 't x u du/dt du/dx d^2u/dt^2 d^2u/dx^2',
             'left_deriv': 'd^2u/dt^2',},
}
