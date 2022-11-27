#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  bool dividable[101];
  for (int i = 0; i < 101; ++i)
    dividable[i] = false;
  for (int i = 2; i <= 50; i += 2)
    for (int j = 2; j <= 50; j += 2)
      dividable[i + j] = true;
  cout << (dividable[n] ? "YES" : "NO") << endl;
  return 0;
}