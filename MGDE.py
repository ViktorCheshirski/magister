import scipy
import random
import numpy as np
import matplotlib.pyplot as plt

sample_size = 100 #объём выборки
w = 0.3 #степень засорения
model = 1 #номер исследуемой модели
tests = 1 #количество испытаний методом Монте-карло
init_approxs = 100 #количество начальных приближений
gamma = 0.5 #параметр робастности
method = 2 #номер исследуемой оценки 1-MLE 2-MGDE 3-GRE 4-BGRE

area = np.zeros((sample_size, 1), int) #вектор номеров выбранных категорий
J = 3  #количество градаций
X_mesh = np.linspace(-1, 1, sample_size) #вектор координаты точек сетки
p = np.zeros((sample_size, J)) #массив вероятностей выбора категорий
p_temp = np.zeros((sample_size, J)) #временный массив вероятностей выбора категорий
teta = np.zeros(4) #истиные параметры модели
koef = np.zeros(4) #коэффициент функции правдоподобия
tau = 0.0 #средний критерий качества Монте-Карло
S = 0.0 #средний критерий качества оценок откликов

def generate_new_data(): 
    """Генерация новой выборки (засорённой)."""
    for i in range(sample_size): 
        M = random.random()
        if (M - (1 - w) < 0):
            Q = random.random()
            ii = 0
            while Q >= 0:
                Q = Q - p[i][ii]
                ii += 1
            area[i] = ii
        else:
            Q = random.random()
            ii = 0
            while Q >= 0:
                Q = Q - 1 / J
                ii += 1
            area[i] = ii
    pass

def ML(a):
    """Логарифм функции правдоподобия."""
    res = 0.
    nu = np.zeros((sample_size, J)) #значение Xij * alpha
    nu[:, 0] = a[0] + X_mesh * a[1]
    nu[:, 1] = a[2] + X_mesh * a[3]
    nu_max = np.max(nu, axis=1, keepdims=True)
    t = nu[np.hstack([(area == 1), (area == 2), (area == 3)])]
    lg = np.log(np.sum(np.exp(nu - nu_max), axis=1))
    res = np.sum(t - nu_max - lg)
    return -res

def get_grad_ml(x):
    """Вычисление градиента функции максимального правдоподобия в точке x"""
    # grad = [0]*4
    # ln_sum = 0.0
    # for i in range (sample_size):
    #     ln_sum += np.exp(x[0] + X_mesh[i] * x[1]) / (np.exp(x[0] + X_mesh[i] * x[1]) + np.exp(x[2] + X_mesh[i] * x[3]) + 1)
    # grad[0] = -koef[0] + ln_sum
    # ln_sum = 0.0
    # for i in range (sample_size):
    #     ln_sum += (X_mesh[i] * np.exp(x[0] + X_mesh[i] * x[1])) / (np.exp(x[0] + X_mesh[i] * x[1]) + np.exp(x[2] + X_mesh[i] * x[3]) + 1)
    # grad[1] = -koef[1] + ln_sum
    # ln_sum = 0.0
    # for i in range (sample_size):
    #     ln_sum += np.exp(x[2] + X_mesh[i] * x[3]) / (np.exp(x[0] + X_mesh[i] * x[1]) + np.exp(x[2] + X_mesh[i] * x[3]) + 1)
    # grad[2] = -koef[2] + ln_sum
    # ln_sum = 0.0
    # for i in range (sample_size):
    #     ln_sum += (X_mesh[i] * np.exp(x[2] + X_mesh[i] * x[3])) / (np.exp(x[0] + X_mesh[i] * x[1]) + np.exp(x[2] + X_mesh[i] * x[3]) + 1)
    # grad[3] = -koef[3] + ln_sum

    # return grad
    return scipy.optimize.approx_fprime(x, ML)

def get_grad_mgd(x):
    """Вычисление приближенного градиента функции минимума гамма-дивергенции в точке x."""
    return scipy.optimize.approx_fprime(x, MGD)

def get_grad_gr(x):
    """Вычисление приближенного градиента обобщённой радикальной функции в точке x."""
    return scipy.optimize.approx_fprime(x, GR)

def get_grad_bgr(x):
    """Вычисление приближенного градиента байесовской обобщённой радикальной функции в точке x."""
    return scipy.optimize.approx_fprime(x, BGR)

def get_rho (a): 
    """Вычисление вероятности в новой точке."""
    nu = np.zeros((sample_size, J)) #значение Xij * alpha
    nu[:, 0] = a[0] + X_mesh * a[1]
    nu[:, 1] = a[2] + X_mesh * a[3]
    nu_max = np.max(nu, axis=1, keepdims=True)
    res = np.exp(nu - nu_max - np.log(np.sum(np.exp(nu - nu_max), axis=1, keepdims=True)))
    return res

def MGD(x):
    """Функция минимума гамма-дивергенции"""
    p_temp = get_rho (x)
    left = p_temp[np.hstack([(area == 1), (area == 2), (area == 3)])]** gamma #вектор вероятностей
    right = np.sum(p_temp**(1+gamma), axis=1)**(-gamma / (1 + gamma)) #вектор дельт
    res = np.sum(left*right)
    return -(res / (sample_size * gamma))

def GR(a):
    """Обобщённая радикальная функция"""
    nu = np.zeros((sample_size, J)) #значение Xij * alpha
    nu[:, 0] = a[0] + X_mesh * a[1]
    nu[:, 1] = a[2] + X_mesh * a[3]
    p_temp = get_rho(a)
    indz = np.hstack([(area == 1), (area == 2), (area == 3)])
    left = p_temp[indz]**gamma #вектор вероятностей
    right = indz.astype(int) - ((p_temp**(1+gamma))/np.sum(p_temp**(1+gamma), axis=1, keepdims=True))#правая квадратная скобка
    der = np.zeros((sample_size, J))
    der[:, 0] = (1 + X_mesh)*(np.exp(nu[:, 1]) + np.exp(nu[:, 2]))
    der[:, 1] = (1 + X_mesh)*(np.exp(nu[:, 0]) + np.exp(nu[:, 2]))
    der[:, 2] = (1 + X_mesh)*(np.exp(nu[:, 0]) + np.exp(nu[:, 1]))
    der = der/np.sum(np.exp(nu), axis=1, keepdims=True)
    right = der*right
    right = np.sum(right, axis=1)
    delta = 1
    res = left*delta*right

    return np.sum(res)

def BGR(a):
    """Байесовская обобщённая радикальная функция"""
    nu = np.zeros((sample_size, J)) #значение Xij * alpha
    nu[:, 0] = a[0] + X_mesh * a[1]
    nu[:, 1] = a[2] + X_mesh * a[3]
    p_temp = get_rho(a)
    indz = np.hstack([(area == 1), (area == 2), (area == 3)])
    left = p_temp[indz]**gamma #вектор вероятностей
    right = indz.astype(int) - ((p_temp**(1+gamma))/np.sum(p_temp**(1+gamma), axis=1, keepdims=True))#правая квадратная скобка
    der = np.zeros((sample_size, J))
    der[:, 0] = (1 + X_mesh)*(np.exp(nu[:, 1]) + np.exp(nu[:, 2]))
    der[:, 1] = (1 + X_mesh)*(np.exp(nu[:, 0]) + np.exp(nu[:, 2]))
    der[:, 2] = (1 + X_mesh)*(np.exp(nu[:, 0]) + np.exp(nu[:, 1]))
    der = der/np.sum(np.exp(nu), axis=1, keepdims=True)
    right = der*right
    right = np.sum(right, axis=1)
    delta = np.sum(p_temp**(1-gamma), axis=1)
    res = left*delta*right

    return np.sum(res)

def montekarlo():
    """Проведение исследований методом Монте-Карло."""
    xk = np.random.uniform(-40, 40, size=(4))
    sum = 0.0
    s = 0.0
    D = 0.0 
    for M in range (tests):
        x_best = np.zeros(4)
        f_best = 0.0
        generate_new_data()
        # find_koef()
        xk = np.random.uniform(-40, 40, size=(4))
        est = scipy.optimize.minimize(ML, xk, method='L-BFGS-B', bounds=[(-80, 80), (-80, 80), (-80, 80), (-80, 80)], jac=get_grad_ml, options={'gtol': 1e-4, 'maxiter': 5000}) #нахождение оценки максимального правдоподобия
        if (method == 1):
            est = est
            f_best = est.fun
            x_best = est.x
            for i in range (1, init_approxs):
                xk = np.random.uniform(-40, 40, size=(4))
                est = scipy.optimize.minimize(ML, xk, method='L-BFGS-B', bounds=[(-80, 80), (-80, 80), (-80, 80), (-80, 80)], jac=get_grad_ml, options={'gtol': 1e-2, 'maxiter': 1000}) #нахождение оценки максимального правдоподобия
                if (f_best > est.fun):
                    f_best = est.fun
                    x_best = est.x
        elif (method == 2):
            est = scipy.optimize.minimize(MGD, est.x, method='L-BFGS-B', bounds=[(-80, 80), (-80, 80), (-80, 80), (-80, 80)], jac=get_grad_mgd, options={'gtol': 1e-4, 'maxiter': 5000}) #нахождение оценки минимума гамма-дивергенции
            f_best = est.fun
            x_best = est.x
            for i in range (1, init_approxs):
                xk = np.random.uniform(-40, 40, size=(4))
                est = scipy.optimize.minimize(MGD, xk, method='L-BFGS-B', bounds=[(-80, 80), (-80, 80), (-80, 80), (-80, 80)], jac=get_grad_mgd, options={'gtol': 1e-4, 'maxiter': 5000}) #нахождение оценки минимума гамма-дивергенции
                if (f_best > est.fun):
                    f_best = est.fun
                    x_best = est.x
        elif (method == 3):
            est = scipy.optimize.minimize(GR, est.x, method='L-BFGS-B', bounds=[(-80, 80), (-80, 80), (-80, 80), (-80, 80)], jac=get_grad_gr, options={'gtol': 1e-4, 'maxiter': 5000}) #нахождение обобщённой радикальной оценки
            f_best = est.fun
            x_best = est.x
            for i in range (1, init_approxs):
                xk = np.random.uniform(-40, 40, size=(4))
                est = scipy.optimize.minimize(GR, xk, method='L-BFGS-B', bounds=[(-80, 80), (-80, 80), (-80, 80), (-80, 80)], jac=get_grad_gr, options={'gtol': 1e-4, 'maxiter': 5000}) #нахождение обобщённой радикальной оценки
                if (f_best > est.fun):
                    f_best = est.fun
                    x_best = est.x
        elif (method == 4):
            est = scipy.optimize.minimize(BGR, est.x, method='L-BFGS-B', bounds=[(-80, 80), (-80, 80), (-80, 80), (-80, 80)], jac=get_grad_bgr, options={'gtol': 1e-4, 'maxiter': 5000}) #нахождение обобщённой радикальной оценки
            f_best = est.fun
            x_best = est.x
            for i in range (1, init_approxs):
                xk = np.random.uniform(-40, 40, size=(4))
                est = scipy.optimize.minimize(BGR, xk, method='L-BFGS-B', bounds=[(-80, 80), (-80, 80), (-80, 80), (-80, 80)], jac=get_grad_bgr, options={'gtol': 1e-4, 'maxiter': 5000}) #нахождение обобщённой радикальной оценки
                if (f_best > est.fun):
                    f_best = est.fun
                    x_best = est.x
        t = np.sqrt(np.sum((x_best-teta)**2))
        D += t**2
        sum += t
        p_temp = get_rho(x_best)
        s += np.sum((p-p_temp)**2)
        print("Mi =", M)
    D = D/tests
    tau = sum / tests
    D -= tau**2
    S = s / sample_size / J
    file = open("out.txt", "a")
    file.write(f"sample_size = {sample_size} w = {w} model = {model} tests = {tests} init_approxs = {init_approxs} gamma = {gamma} method = {method}\n")
    file.write(f"S = {S}\n")
    file.write(f"Tau = {tau}\n")
    file.write(f"D = {D}\n\n")
    file.close()
    pass

def montekarlo2():
    """Проведение исследований методом Монте-Карло."""
    sum = 0.0
    s = 0.0
    D = 0.0 
    for M in range (tests):
        x_best = np.zeros(4)
        f_best = 0.0
        generate_new_data()
        if (method == 1):
            est = scipy.optimize.minimize(ML, teta, method='L-BFGS-B', bounds=[(-80, 80), (-80, 80), (-80, 80), (-80, 80)], jac=get_grad_ml, options={'gtol': 1e-4, 'maxiter': 5000}) #нахождение оценки минимума гамма-дивергенции
            x_best = est.x
        elif (method == 2):
            est = scipy.optimize.minimize(MGD, teta, method='L-BFGS-B', bounds=[(-80, 80), (-80, 80), (-80, 80), (-80, 80)], jac=get_grad_mgd, options={'gtol': 1e-4, 'maxiter': 5000}) #нахождение оценки минимума гамма-дивергенции
            x_best = est.x
        elif (method == 3):
            est = scipy.optimize.minimize(GR, teta, method='L-BFGS-B', bounds=[(-80, 80), (-80, 80), (-80, 80), (-80, 80)], jac=get_grad_gr, options={'gtol': 1e-4, 'maxiter': 5000}) #нахождение обобщённой радикальной оценки
            x_best = est.x
        elif (method == 4):
            est = scipy.optimize.minimize(BGR, teta, method='L-BFGS-B', bounds=[(-80, 80), (-80, 80), (-80, 80), (-80, 80)], jac=get_grad_bgr, options={'gtol': 1e-4, 'maxiter': 5000}) #нахождение обобщённой радикальной оценки
            x_best = est.x
        t = np.sqrt(np.sum((x_best-teta)**2))
        D += t**2
        sum += t
        p_temp = get_rho(x_best)
        s += np.sum((p-p_temp)**2)
    D = D/tests
    tau = sum / tests
    D -= tau**2
    S = s / sample_size / J
    file = open("out.txt", "a")
    file.write(f"sample_size = {sample_size} w = {w} model = {model} tests = {tests} init_approxs = {init_approxs} gamma = {gamma} method = {method}\n")
    file.write(f"S = {S}\n")
    file.write(f"Tau = {tau}\n")
    file.write(f"D = {D}\n")
    file.close()
    pass

if __name__ == "__main__":

    if (model == 1):
        teta = np.array([-10, 20, -8, -20])
    elif (model == 2):
        teta = np.array([-1.5, -2.5, 0, 0.5])
    p = get_rho(teta)

    # w = 0.0 #степень засорения
    # tests = 100 #количество испытаний методом Монте-карло
    # init_approxs = 100 #количество начальных приближений
    # gamma = 0.4 #параметр робастности
    # method = 1 #номер исследуемой оценки 1-MLE 2-MGDE 3-GRE 4-BGRE
    # montekarlo()

    # w = 0.05 #степень засорения
    # montekarlo()

    # w = 0.1 #степень засорения
    # montekarlo()

    # w = 0.20 #степень засорения
    # montekarlo()

    # w = 0.3 #степень засорения
    # montekarlo()



    generate_new_data()
    # xk = [-10, 20, -8, -20]
    # find_koef()
    # MLE = scipy.optimize.minimize(ML, xk, method='BFGS', jac=get_grad_ml) #нахождение оценки максимального правдоподобия
    # MGDE = scipy.optimize.minimize(MGD, xk, method='BFGS', jac=get_grad_mgd) #нахождение оценки минимума гамма-дивергенции
    # GRE = scipy.optimize.minimize(GR, xk, method='BFGS', jac=get_grad_gr) #нахождение обобщённой радикальной оценки
    # BGRE = scipy.optimize.minimize(BGR, xk, method='BFGS', jac=get_grad_bgr) #нахождение обобщённой радикальной оценки

    #графики
    fig, ax = plt.subplots()

    # y1 = [0]*sample_size
    # y2 = [0]*sample_size
    # y3 = [0]*sample_size

    # p_temp = get_rho(MGDE.x)
    # for i in range (sample_size):
    #     y1[i] = p_temp[i][0]
    #     y2[i] = p_temp[i][1]
    #     y3[i] = p_temp[i][2]
    # ax.plot(X_mesh, p_temp)
    # ax.plot(X_mesh, y2)
    # ax.plot(X_mesh, y3)
    # plt.ylabel('Вероятности')
    # plt.xlabel('Значения фактора')
    # plt.legend(['P(z=1|x)', 'P(z=2|x)', 'P(z=3|x)'])

    ax.plot(X_mesh, area, '.')
    plt.yticks(np.arange(min(area), max(area)+1, 1.0))  # изменяем шаг делений на оси X
    plt.ylabel('Градации')
    plt.xlabel('Значения фактора')

    plt.show()