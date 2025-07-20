#include <bits/stdc++.h>
using namespace std;
void solve() {
  string s;
  cin >> s;
  int index = s.find("h");
  if (index >= 0 && index <= s.length()) {
    int dem = 1;
    string ss = s.substr(index);
    int n = ss.length();
    for (int i = 0; i < n; i++) {
      if (ss[i] == 'h' && dem == 0) {
        dem++;
      }
      if (ss[i] == 'e' && dem == 1) {
        dem++;
      }
      if (ss[i] == 'l' && dem < 5 && dem >= 2) {
        dem++;
      }
      if (ss[i] == 'o' && dem == 4) {
        dem++;
      }

      //  cout<<ss[i]<<" "<<dem<<endl;
      if (dem == 5) {
        cout << "YES";
        exit(0);
      }
    }
    cout << "NO";
  } else {
    cout << "NO";
  }
  // wfluaeseldgxyvxpwuhkptdmlfln
  // lhktwxiabsvkolsquymrmhzczzoybvlilhmvsuunggvgxzgyyffk
}
int main() {
  solve();

  return 0;
}
