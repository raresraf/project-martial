#include <iostream>
using namespace std;

int main() {
  int w;
  cin >> w;
  if (w % 2 != 0 || w < 3)
    cout << "NO";
  else
    cout << "YES";
  return 0;
}