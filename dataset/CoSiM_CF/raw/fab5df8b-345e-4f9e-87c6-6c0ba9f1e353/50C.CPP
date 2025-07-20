#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
#define ten5 100000 + 10
#define MOD 1000000007
#define rep(i, a, n) for (int i = a; i < n; i++)
#define iif(c, t, f) ((c) ? (t) : (f))
#define per(i, a, n) for (int i = n - 1; i >= a; i--)
#define pb push_back
#define mp make_pair

int intcmp(const void *v1, const void *v2) { return *(int *)v1 - *(int *)v2; }

// < 10000000 ,3 = len=5
// len =n 1000000 ,4 ->  (n-1,5)

// len = n, number of 'not zero' = cnt
ll partCnt(int n, int cnt) {
  if (cnt > n)
    return 0;
  ll ret = 1;
  rep(i, 0, cnt) { ret *= 9; }
  // ret*C(n-1,cnt);
  rep(i, 0, cnt) { ret *= n - i; }
  rep(i, 0, cnt) { ret /= i + 1; }
  // cout<<"part"<<n<<","<<cnt<<"\t"<<ret<<endl;
  return ret;
}
// len <= n , 'not zero' <= cnt
ll allCnt(int n, int cnt) {
  ll ret = 0;
  rep(i, 0, cnt + 1) { ret += partCnt(n, i); }
  return ret;
}

int vdiv10[20];
const int N = 3;
ll q(ll v) {
  // cout<<"Q "<<v<<endl;
  int length = 0;
  ll vv = v;
  while (vv != 0) {
    vdiv10[length] = vv % 10;
    vv /= 10;
    length++;
  }
  int no0cnt = 0;
  ll ret = 0;
  for (int i = length - 1; i >= 0; i--) {
    if (vdiv10[i] == 0)
      continue;
    no0cnt++;
    if (no0cnt > N)
      continue;
    if (no0cnt == 1) {
      ll der = (vdiv10[i] - 1) * allCnt(i, N - no0cnt) + 1;
      // cout<<no0cnt<<":"<<der<<endl;
      ret += der;
    } else {
      ll der =
          (vdiv10[i] - 1) * allCnt(i, N - no0cnt) + allCnt(i, N - no0cnt + 1);
      // cout<<no0cnt<<":"<<der<<endl;
      ret += der;
    }
  }
  for (int i = length - 2; i >= 0; i--) {
    ret += 9 * allCnt(i, N - 1);
  }
  return ret;
}

int main() {
  // rep(i,0,7){
  //  int  allcnt = allCnt(i,i);
  //  cout<<i<<"\t"<<allcnt<<endl;
  //}
  int t;
  cin >> t;
  while (t-- > 0) {
    ll l, r;
    scanf("%lld %lld", &l, &r);
    printf("%lld\n", q(r) - q(l - 1));
  }
  return 0;
}
