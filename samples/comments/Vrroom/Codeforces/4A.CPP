#include <iostream>

using namespace std;

int main() {
  int W;
  cin >> W;
  if (W != 2 && W % 2 == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
