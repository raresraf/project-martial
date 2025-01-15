#include <iostream>
using namespace std;

typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  ios_base::sync_with_stdio(false);
  lli t, a, b;
  cin >> t;
  lli sum = 0;
  lli max = 0;
  while (t--) {
    cin >> a >> b;
    sum -= a;
    sum += b;
    sum >= max ? max = sum : max = max;
  }
  cout << max << endl;
  return 0;
}
