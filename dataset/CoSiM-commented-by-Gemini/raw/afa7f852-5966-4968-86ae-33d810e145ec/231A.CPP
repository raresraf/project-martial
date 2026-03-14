/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @see https://codeforces.com/problemset/problem/231/A
 *
 * This program counts the number of problems a team will solve. A problem is
 * solved if at least two out of the three team members are sure about the solution.
 */
/*==============================================*
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
#define pi(a) printf("%d
", a);
#define ppi(i, a) printf("Case %d: %d
", i, a);
#define ll long long
#define fast                                                                   
  ios_base::sync_with_stdio(false);                                            
  cin.tie(NULL);                                                               
  cout.tie(NULL);
void FI() {
#ifndef ONLINE_JUDGE
  freopen("C:\Users\SHIHAB\Desktop\input.in", "r", stdin);
#endif // ONLINE_JUDGE
}

int main() {
  fast FI(); /// skip this line for compile
  int n, a, b, c, cnt = 0;
  cin >> n;
  while (n--) {
    cin >> a >> b >> c;
    /**
     * @brief Check if at least two team members are sure.
     * The inputs a, b, and c are either 0 or 1. A sum greater than 1
     * means that at least two of them are 1.
     */
    if (a + b + c > 1)
      cnt++;
  }
  cout << cnt << endl;
  return 0;
}
