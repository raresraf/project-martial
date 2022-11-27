#include <iostream>
using namespace std;

int main() {
  int w;
  cin >> w;
  w -= 2;
  if (!(w % 2) && w > 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
