/*==============================================*\
Codeforces ID:    mdshihab
                                                  |
Name:     Md. Shihabul Minhaz                     |
Study:      CSE, JKKNIU                           |
Address:  Trishal, Mymensingh, Bangladesh         |
                                                  |
 mail:   mdshihabul20171@gmail.com                |
 FB  :   fb.com/mdshihabul.minhaz.7               |
 github: Shihabulminhaz                           |
stopstalk :  mdshihab                             |
                                                  |
@uthor   Md. Shihabul Minhaz (shihab)             |
\*===============================================*/

#include <bits/stdc++.h>
using namespace std;
#define fi(a) scanf("%d", &a);
#define fli(a) scanf("%ld", &a);
#define fll(a) scanf("%lld", &a);
#define pi(a) printf("%d\n", a);
#define ppi(i, a) printf("Case %d: %d\n", i, a);
#define ll long long
#define fast                                                                   \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);
void FI() {
#ifndef ONLINE_JUDGE
  freopen("C:\\Users\\SHIHAB\\Desktop\\input.in", "r", stdin);
#endif // ONLINE_JUDGE
}

int main() {
  fast
      // FI(); /// skip this line for compile
      int n,
      cnt = 0;
  string s;
  cin >> n;
  while (n--) {
    cin >> s;
    if (s[0] == '-' || s[2] == '-')
      cnt--;
    else
      cnt++;
  }
  cout << cnt << endl;
  return 0;
}
