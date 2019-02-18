workflow "on push" {
  on = "push"
  resolves = ["Python Syntax Checker"]
}

action "Python Syntax Checker" {
  uses = "cclauss/Find-Python-syntax-errors-action@master"
}
