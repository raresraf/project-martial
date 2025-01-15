#include <iostream>
using namespace std;

typedef long long int lli;
typedef long long ll;

int main(int argc, char const *argv[]) {
  ios_base::sync_with_stdio(false);
  lli t;
  cin >> t;
  while (t--) {
    string a;
    cin >> a;
    if (a.size() > 10) {
      cout << a[0] << +a.size() - 2 << a[a.size() - 1] << endl;
    } else {
      cout << a << endl;
    }
  }
  return 0;
}
