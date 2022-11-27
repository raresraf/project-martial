#include <iostream>
using namespace std;

int main() {
  int a;
  cin >> a;
  int t = a % 2;
  if (a == 2)
    t = 1;
  if (t)
    cout << "NO" << endl;
  else
    cout << "YES" << endl;
  return 0;
}