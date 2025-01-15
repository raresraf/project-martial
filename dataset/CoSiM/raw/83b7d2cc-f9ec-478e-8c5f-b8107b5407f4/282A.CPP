#include <iostream>
using namespace std;

int main() {
  int num, x = 0;
  string statement;
  cin >> num;
  while (num--) {
    cin >> statement;
    if (statement.find("++") != -1) // string::npos==-1
      x++;
    if (statement.find("--") != -1)
      x--;
  }
  cout << x << endl;
  return 0;
}
