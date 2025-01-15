#include <iostream>
using namespace std;
int main() {
  long long int n;
  cin >> n;
  long int ans, rem;
  ans = n / 7;
  rem = n % 7;
  if (rem <= 2)
    cout << ans * 2 << " " << ans * 2 + rem;
  else {
    if (rem == 6)
      cout << ans * 2 + 1 << " " << ans * 2 + 2;
    else
      cout << ans * 2 << " " << ans * 2 + 2;
  }
}