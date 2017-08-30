import tensorflow as tf
#from utils.expLogger improt expLogger

def metropolisHastingsAccept(energyPre,eneryNext,expLog,ifuseLogger = False):# TODO: maybe move this to the utils folder
    """
    Run Metropolis-Hastings algorithm for 1 step
    :param energyPrev: the original energy from before
    :param energyNext: the energy after evolution
    :return: Tensor of boolean values, indicating accept or reject
    """
    energyDiff = energyPre - eneryNext
    if ifuseLogger:
        return expLog.cal(energyDiff) # TODO: The energyDiff is a tf tensor object, need some workaround
    return (tf.exp(energyDiff)) - tf.random_uniform(tf.shape(energyPre)) >= 0.0