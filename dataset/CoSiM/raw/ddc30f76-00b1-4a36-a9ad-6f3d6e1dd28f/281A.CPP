#include <iostream>
using namespace std;

int main() {
  char c[1000];
  cin >> c;
  if (c[0] >= 97)
    c[0] -= 32;
  cout << c << endl;

  return 0;
}
