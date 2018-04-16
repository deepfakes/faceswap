import sys
import locale

system_locale = locale.getdefaultlocale()[0]
system_language = system_locale[0:2]

windows_font_name_map = {
    'en' : 'cour',
    'ru' : 'cour',
    'zn' : 'simsun_01'
}

darwin_font_name_map = {
    'en' : 'cour',
    'ru' : 'cour',
    'zn' : 'Apple LiSung Light'
}

linux_font_name_map = {
    'en' : 'cour',
    'ru' : 'cour',
    'zn' : 'cour'
}

def get_default_ttf_font_name():
    platform = sys.platform
    if platform == 'win32': return windows_font_name_map.get(system_language, 'cour')
    elif platform == 'darwin': return darwin_font_name_map.get(system_language, 'cour')
    else: return linux_font_name_map.get(system_language, 'cour')
