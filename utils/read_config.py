from configparser import ConfigParser
import os


def config(section, option_key, file_name='../resource/config.ini'):
    """读取配置文件
    读取位于项目根目录下的配置文件，默认配置文件名为 config.ini
    :param section: ini 文件中的 section
    :param option_key: ini 文件中的 option 项
    :param file_name: 配置文件名，默认为 config.ini
    :return: 返回 ini 文件中指定的 option 值
    """
    cur_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_path, file_name)
    cp = ConfigParser()
    cp.read(config_path)
    value = cp.get(section, option_key)
    return value


if __name__ == '__main__':
    pass
