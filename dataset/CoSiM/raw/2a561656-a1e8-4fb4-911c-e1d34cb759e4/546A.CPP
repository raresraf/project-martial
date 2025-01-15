/*=====TENSHI29=====*/
#include <iostream>
using namespace std;
int main() {
  int k, n, w, sum;
  cin >> k >> n >> w;
  sum = w * (1 + w);
  sum = sum / 2;
  sum = sum * k;
  if (sum - n < 0)
    cout << '0' << endl;
  else
    cout << sum - n << endl;
  return 0;
}
