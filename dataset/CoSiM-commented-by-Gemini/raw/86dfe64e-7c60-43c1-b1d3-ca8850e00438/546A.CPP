#include <iostream>
using namespace std;

int main() {

  int k, n, w;
  int amt = 0;
  cin >> k >> n >> w;

  for (int i = 1; i <= w; i++)
    amt += i * k;

  if (amt - n > 0)
    cout << amt - n << endl;
  else
    cout << 0 << endl;
  return 0;
}
