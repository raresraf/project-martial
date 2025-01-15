#include <algorithm>
#include <cstring>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace std;

typedef long long int ll;
typedef pair<int, int> pii;
#define pb push_back
#define mp make_pair
#define PI acos(-1.0)

// Normal Gcd
/*int gcd ( int a, int b )
{
  int c;
  while ( a != 0 ) {
     c = a; a = b%a;  b = c;
  }
  return b;
}*/

// Recursive Gcd
/*int gcdr ( int a, int b )
{
  if ( a==0 ) return b;
  return gcdr ( b%a, a );
}*/

// moduler arithmetic
// ll M
/*ll bigmod(ll B,ll P)
{
    if(P==0)return 1;
    if(P%2==0)
    {
        ll ret=(bigmod(B,P/2))%M;
        return ((ret%M)*(ret%M))%M;
    }
    else
        return ((B%M)*(bigmod(B,P-1)%M))%M;
}*/

// int fx[]={0,0,1,-1};
// int fy[]={1,-1,0,0};
// int fx[]={0,0,1,1,1,-1,-1,-1};
// int fy[]={1,-1,0,1,-1,0,1,-1};
// int fx[]={1,1,-1,-1,2,-2,2,-2};
// int fy[]={2,-2,2,-2,1,1,-1,-1};

int main() {
  ll n, mini, maxi = 2;
  cin >> n;
  mini = 2 * (n / 7);
  if (n % 7 > 5)
    mini += 1;
  if (n > 2)
    maxi += 2 * ((n - 2) / 7);
  else
    maxi = n;
  if ((n - 2) % 7 > 5)
    maxi += 1;
  cout << mini << " " << maxi << endl;
  return 0;
}
