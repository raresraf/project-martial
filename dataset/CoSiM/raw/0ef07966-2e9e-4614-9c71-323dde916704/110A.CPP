//--------------**************---------------------
/*
 #        "    ""#    ""#                      #
 #   m  mmm      #      #     mmm    mmm    mmm#   mmm
 # m"     #      #      #    #"  "  #" "#  #" "#  #"  #
 #"#      #      #      #    #      #   #  #   #  #""""
 #  "m  mm#mm    "mm    "mm  "#mm"  "#m#"  "#m##  "#mm"  */

#include <bits/stdc++.h>
using namespace std;
#define pb push_back
#define mp make_pair
typedef pair<int, int> pii;
typedef long long ll;
typedef double ld;
typedef vector<int> vi;
#define fi first
#define se second
#define fe first
#define SZ 666666
#define si(n) scanf("%d", &n);
#define sl(n) scanf("%ld", &n);
#define pi(n) printf("%d\n", n);
#define pl(n) printf("%ld\n", n);
#define pf(n) printf("%f\n", n);
#define FILL(a, b) memset(a, 0, sizeof(b));
#define rep(i, n) for (int i = 0; i < n; i++)
#define reps(i, a, b) for (int i = 1; i <= b; i++)
const int INF = 1e9 + 5;
const int MOD = 1000000007;

//--------------**************---------------------

int main() {
  ll n;
  cin >> n;
  bool flag = false;
  int count = 0;
  while (n > 0) {
    if (n % 10 == 7 || n % 10 == 4)
      count++;
    n /= 10;
  }
  if (count == 0) {
    cout << "NO";
    return 0;
  }

  if (count == 4 || count == 7)
    flag = true;
  if (flag)
    cout << "YES";
  else
    cout << "NO";
}
