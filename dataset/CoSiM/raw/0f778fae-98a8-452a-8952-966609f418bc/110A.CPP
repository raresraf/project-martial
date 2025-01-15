/*
    Problem Link    :   https://codeforces.com/problemset/problem/110/A
    Contest Link    :   https://codeforces.com/contest/110/problem/A
    Status          :   Accepted

    Author          :   Nikhil Nagrale
    Codeforces      :   https://codeforces.com/profile/nikhilnagrale2
    Codechef        :   https://www.codechef.com/users/nikhilnagrale2
    Github          :   https://github.com/nikhilnagrale2

    Copyright       :   Solve it on your own.
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
  long long n;
  cin >> n;
  long long count = 0;
  while (n != 0) {
    int x = n % 10;
    n /= 10;
    if (x == 4 || x == 7) {
      count++;
    }
  }

  if (count == 0) {
    cout << "NO" << endl;
    return 0;
  }

  bool flag = true;
  while (count != 0) {
    int x = count % 10;
    count /= 10;
    if (x != 4 && x != 7) {
      flag = false;
      cout << "NO" << endl;
      break;
    }
  }

  if (flag)
    cout << "YES" << endl;
  return 0;
}

/*
smaller lines of solution
int count=0;
for(int i=0;i<s.size();i++)
{
    if(s[i]=='4' || s[i]=='7) count++;
}
if(w==4 || w==7) cout<<"YES"<<endl;
else cout<<"NO"<<endl;
*/