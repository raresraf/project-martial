#include <iostream>
using namespace std;

typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;
typedef double dbl;
typedef string str;

int main(int argc, char const *argv[]) {
  ios_base::sync_with_stdio(false);
  str a;
  cin >> a;
  lli count = 0;
  for (int i = 0; i < a.size(); ++i) {
    if (a[i] == '4' || a[i] == '7')
      count++;
  }
  if (((count == 4) || (count == 7)) && (count > 0))
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}