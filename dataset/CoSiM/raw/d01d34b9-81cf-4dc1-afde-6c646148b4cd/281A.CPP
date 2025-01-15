#include <bits/stdc++.h>
#include <cstring>
#include <string>
using namespace std;
int t, jercy;
char card, team;
int home[101], away[101];
int main() {
  string hm, ay;
  int n;
  cin >> hm >> ay;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> t >> team >> jercy >> card;
    if (card == 'r') {
      if (team == 'h') {
        if (home[jercy] == -1)
          continue;
        cout << hm << " " << jercy << " " << t << endl;
        home[jercy] = -1;
      } else {
        if (away[jercy] == -1)
          continue;
        cout << ay << " " << jercy << " " << t << endl;
        away[jercy] = -1;
      }

    }

    else {
      if (team == 'h') {
        if (home[jercy] == -1)
          continue;
        else if (home[jercy] == 1) {
          cout << hm << " " << jercy << " " << t << endl;
          home[jercy] = -1;
        } else
          home[jercy] = 1;
      } else {
        if (away[jercy] == -1)
          continue;
        else if (away[jercy] == 1) {
          cout << ay << " " << jercy << " " << t << endl;
          away[jercy] = -1;
        } else
          away[jercy] = 1;
      }
    }
  }

  return 0;
}
