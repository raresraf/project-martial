#include <iostream>
using namespace std;
int main() {
  char c;
  int num = 0;
  while (cin >> c) {
    if (c == '4' || c == '7')
      num++;
  }
  cout << ((num == 4 || num == 7) ? "YES" : "NO");
  return 0;
}
