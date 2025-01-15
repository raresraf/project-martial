///       From dust I came and to dust I shall return
/**
                    Asadullah Bin Rahman
                          CSE-17
    HAJEE MOHAMMAD DANESH SCIENCE AND TECHNOLOGY UNIVERSITY
*/
#include <bits/stdc++.h>
#define sf(f) scanf("%d", &f)
#define pf(f) printf("%d\n", f)
#define PI 3.1415926535897932385
#define size 3000000
#define mp make_pair
#define pb push_back
#define one(x) __builtin_popcount(x)     /// int
#define onel(x) __builtin_popcountl(x)   /// long int
#define onell(x) __builtin_popcountll(x) /// long long
using namespace std;
typedef long long ll;

/// int dx[]= {-1,-1,0,0,1,1};
/// int dy[]= {-1,0,-1,1,0,1};
/// int dx[]= {0,0,1,-1};                 /*4 side move*/
/// int dy[]= {-1,1,0,0};                 /*4 side move*/
/// int dx[]= {1,1,0,-1,-1,-1,0,1};       /*8 side move*/
/// int dy[]= {0,1,1,1,0,-1,-1,-1};       /*8 side move*/
/// int dx[]={1,1,2,2,-1,-1,-2,-2};       /*knight move*/
/// int dy[]={2,-2,1,-1,2,-2,1,-1};       /*knight move*/
/// Always Remember   'A'=65,'Z'=90 'a'=97 'z'=122 '0'=48

int main() {
  string s;
  cin >> s;
  int count = 1;
  int l = s.length();
  for (int i = 1; i < l; ++i) {
    if (s[i] == s[i - 1]) {
      // cout<<"fuck\n";
      count += 1;
      if (count == 7) {
        cout << "YES" << endl;
        return 0;
      }
    }

    else
      count = 1;
  }

  cout << "NO" << endl;
  return 0;
}
