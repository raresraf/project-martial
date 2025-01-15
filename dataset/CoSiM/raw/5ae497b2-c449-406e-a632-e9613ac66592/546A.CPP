#include <iostream>
using namespace std;
int main() {
  long long int k, n, w;
  cin >> k >> n >> w;
  long long int mult;
  mult = (k * w * (w + 1)) / 2;
  mult = mult - n;
  if (mult >= 0)
    cout << mult << endl;
  else
    cout << "0" << endl;
  return 0;
}
