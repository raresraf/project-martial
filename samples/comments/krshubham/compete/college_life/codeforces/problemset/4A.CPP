#include <iostream>
using namespace std;

typedef long long int lli;

int main(int argc, char const *argv[]) {
  ios_base::sync_with_stdio(false);
  lli a;
  cin >> a;
  if (a <= 2) {
    cout << "NO" << endl;
    return 0;
  }
  (a % 2 == 0) ? cout << "YES" << endl : cout << "NO" << endl;
  return 0;
}
