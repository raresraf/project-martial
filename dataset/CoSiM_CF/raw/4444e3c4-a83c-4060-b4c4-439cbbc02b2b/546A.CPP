/**
 * I'm sorry I was an Idiot
 */
#include <iostream>
#define ll long long
#define max(x, y) (x > y ? x : y)
using namespace std;

int main() {
  ll k, n, w;
  cin >> k >> n >> w;
  ll cost = k * ((w * (w + 1)) >> 1);
  cout << max((cost - n), 0) << endl;
  return 0;
}