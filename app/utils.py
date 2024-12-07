class SimpleLogger:
  level = 0

  def debug(self, *params):
    if self.level == 0:
      print('\033[1m[DEBUG]\033[0m:', *params)

  def info(self, *params):
    if self.level <= 1:
      print('\033[96m\033[1m[INFO]\033[0m:', *params)

  def warning(self, *params):
    if self.level <= 2:
      print('\033[93m\033[1m[WARNING]\033[0m:', *params)

  def error(self, *params):
    if self.level <= 3:
      print('\033[31m\033[1m[ERROR]\033[0m:', *params)

  def set_level(self, level=0):
    self.level = level