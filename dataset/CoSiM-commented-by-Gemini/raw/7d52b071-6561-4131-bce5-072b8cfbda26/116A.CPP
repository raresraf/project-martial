/*#include<bits/stdc++.h>
typedef long long int ll;
const ll maxy=1e5+5;
#define rep(i,a,b) for(ll i=a;i<b;i++)
using namespace std;
ll sz[maxy];
void fill(){
    memset(sz,0,sizeof(sz));
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    ll pre;bool ok=0;
    cin>>pre;
    ll p=pre-1;while(p--){
        ll res;cin>>res;sz[res]++;
    } ll ans=0;
    rep(i,0,maxy){
        if(sz[i]>0 and sz[i]>=3){
            continue;
        }else if(sz[i]>0 and sz[i]<=2){ok=1;break;}
    }if(ok){cout<<"No"<<endl;}else{cout<<"Yes"<<endl;}
    return 0;
}*/
#include <bits/stdc++.h>
#define elif else if
#define pb push_back
#define mp make_pair
#define f first
#define second
typedef long long int ll;
#define rep(i, a, b) for (ll i = a; i < b; i++)
using namespace std;
void solve() {
  ll count = 0;
  ll sum = 0;
  ll query;
  cin >> query;
  ll res = query - 1;
  ll a, b;
  cin >> a >> b;
  ll ans = INT_MIN;
  sum += a + b;
  ans = max(ans, sum);
  while (res--) {
    ll p, q;
    cin >> p >> q;
    sum -= p;
    sum += q;
    ans = max(ans, sum);
  }
  cout << ans << endl;
}
int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);
  ll t = 1;
  // cin>>t;
  while (t--) {
    solve();
  }
  return 0;
}