/**
* Git >> Mohsin_Riad


















**/
#include <bits/stdc++.h>
using namespace std;

#define IOS                                                                    \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define mx 1000
#define endl "\n"
#define pb push_back
#define pob pop_back
typedef long long ll;
typedef double dd;
const int N = 1e6;

int main() {
  IOS

      int x[6][6],
      temp;

  for (int i = 1; i <= 5; i++)
    for (int j = 1; j <= 5; j++) {
      cin >> x[i][j];
      if (x[i][j] == 1) {
        if (i == 3 && j == 3)
          temp = 0;
        else if ((i == 1 && j == 1) || (i == 5 && j == 5) ||
                 (i == 1 && j == 5) || (i == 5 && j == 1))
          temp = 4;
        else if ((i == 1 && j == 2) || (i == 2 && j == 1) ||
                 (i == 1 && j == 4) || (i == 4 && j == 1) ||
                 (i == 5 && j == 2) || (i == 2 && j == 5) ||
                 (i == 5 && j == 4) || (i == 4 && j == 5))
          temp = 3;
        else if ((i == 1 && j == 3) || (i == 3 && j == 1) ||
                 (i == 2 && j == 2) || (i == 4 && j == 4) ||
                 (i == 5 && j == 3) || (i == 3 && j == 5) ||
                 (i == 2 && j == 4) || (i == 4 && j == 2))
          temp = 2;
        if ((i == 2 && j == 3) || (i == 3 && j == 2) || (i == 4 && j == 3) ||
            (i == 3 && j == 4))
          temp = 1;
      }
    }
  cout << temp << endl;
  return (0);
}
