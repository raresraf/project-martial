#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string.h>

using namespace std;
typedef long long ll;

int main() {
  // ifstream fi;
  // fi.open("probA.txt");
  // if(!fi){
  //     cout << "Cant open this file";
  //     return 0;
  // }
  ll t;
  cin >> t;
  while (t--) {
    ll sti, di;
    cin >> sti >> di;
    ll ans;
    if (sti == 0 || di == 0) {
      ans = 0;
    } else {
      ans = (sti + di) / 3;
      if (ans > sti || ans > di) {
        ans = min(sti, di);
      }
    }
    cout << ans << endl;
  }
}