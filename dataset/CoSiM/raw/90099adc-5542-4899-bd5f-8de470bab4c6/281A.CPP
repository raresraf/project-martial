#include <iostream>
#include <set>
using namespace std;

int main() {

  set<int> home, away, hdone, adone;
  string hh, aa;
  cin >> hh >> aa;

  int n;
  int t, m;
  string a, b;
  cin >> n;
  while (n--) {
    cin >> t >> a >> m >> b;
    //		cout<<t<<" "<<a<<" "<<m<<" "<<b<<endl;
    if (a == "h") {
      if ((home.count(m) > 0 || b == "r") && hdone.count(m) == 0) {
        cout << hh << " " << m << " " << t << endl;
        hdone.insert(m);
      }
      home.insert(m);
    } else {
      if ((away.count(m) > 0 || b == "r") && adone.count(m) == 0) {
        cout << aa << " " << m << " " << t << endl;
        adone.insert(m);
      }
      away.insert(m);
    }
  }
}