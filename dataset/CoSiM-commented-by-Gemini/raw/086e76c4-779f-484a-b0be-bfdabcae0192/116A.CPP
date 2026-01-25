#include <iostream>
using namespace std;
int main() {
  long int S, exit, enter, rev = 0, tem = 0;
  cin >> S;
  while (S--) {
    cin >> exit >> enter;
    rev = rev + enter - exit;
    tem = tem >= rev ? tem : rev;
  }
  cout << tem;
  return 0;
}
