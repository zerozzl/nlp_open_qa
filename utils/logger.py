from datetime import datetime
import codecs
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


class Logger:
    def __init__(self, data_path=''):
        self.data_path = data_path

        if self.data_path != '':
            with codecs.open('%s/log.txt' % self.data_path, 'w', 'utf-8') as fout:
                fout.write('time\tepoch\tloss\tf1\tem\tremark\n')

    def get_timestamp(self, format='%Y-%m-%d %H:%M:%S'):
        return datetime.strftime(datetime.now(), format)

    def write(self, epoch, loss, f1, em, remark=''):
        with codecs.open('%s/log.txt' % self.data_path, 'a', 'utf-8') as fout:
            fout.write('%s\t%s\t%.5f\t%.3f\t%.3f\t%s\n' % (
                self.get_timestamp(), epoch, loss, f1, em, remark))

    def draw_plot(self, data_path=''):
        if data_path == '':
            data_path = self.data_path

        eppch = []
        loss = []
        f1 = []
        em = []

        with codecs.open('%s/log.txt' % data_path, 'r', 'utf-8') as fin:
            _ = fin.readline()
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                eppch.append(int(line[1]) - 1)
                loss.append(float(line[2]))
                f1.append(float(line[3]))
                em.append(float(line[4]))

        x_locator = MultipleLocator(int(len(eppch) / 5))
        y_locator = MultipleLocator(int(len(eppch) / 10))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        ax = plt.subplot2grid((2, 2), (0, 0), colspan=2, title='loss')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, loss)

        ax = plt.subplot2grid((2, 2), (1, 0), title='f1')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, f1)

        ax = plt.subplot2grid((2, 2), (1, 1), title='em')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, em)

        plt.rcParams['savefig.dpi'] = 200
        plt.savefig('%s/plot.jpg' % data_path)

# if __name__ == '__main__':
#     logger = Logger()
#     logger.draw_plot(data_path='../runtime/bert/ctrip')
