#include <iostream>
using namespace std;

int main() {
  int w{0};
  cin >> w;
  if (w > 2 && w % 2 == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
