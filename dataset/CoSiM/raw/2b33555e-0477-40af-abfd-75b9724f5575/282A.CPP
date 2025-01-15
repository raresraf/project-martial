#include <iostream>
using namespace std;
int main() {
  int n;
  int count = 0;
  string oper;
  cin >> n;
  while (n--) {
    cin >> oper;
    if (oper[0] == '+' || oper[2] == '+')
      count += 1;
    else
      count -= 1;
  }
  cout << count << endl;
  return 0;
}
