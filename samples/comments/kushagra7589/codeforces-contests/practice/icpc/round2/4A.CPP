#include <bits/stdc++.h>

using namespace std;

#define ll long long
#define vi vector<int>
#define vll vector<ll>
#define pii pair<int, int>
#define pli pair<ll, int>
#define pil pair<int, ll>
#define pb push_back
#define mp make_pair
#define F first
#define S second

const int MAX_SIZE = 1e5 + 10;

vector<pli> dis;
vll arr(MAX_SIZE);
vi status(MAX_SIZE);

int main() {
  ios_base::sync_with_stdio(false);
  int n;
  cin >> n;
  for (int i = 1; i <= n; i++) {
    cin >> arr[i];
    dis.pb(mp(arr[i], i));
  }
  sort(dis.begin(), dis.end());
  stack<int> s;
  for (int i = 0; i < dis.size() / 2; i++) {
    s.push(dis[i].S);
    status[dis[i].S] = 1;
  }
  for (int i = dis.size() / 2; i < dis.size(); i++) {
    status[dis[i].S] = 2;
  }
  int i = 1;
  ll jack = 0, other = 0;
  while (i <= n) {
    if (status[i] != 0) {
      if (i == n) {
        jack += arr[i];
        break;
      }
      int u = i;
      int v = i + 1;
      while (v <= n && status[v] == 0)
        v++;
      if (status[u] == 2) {
        if (status[v] == 2) {
          jack += max(arr[u], arr[v]);
          other += min(arr[u], arr[v]);
          status[u] = 0;
          status[v] = 0;
          while (!s.empty() && status[s.top()] == 0)
            s.pop();
          status[s.top()] = 2;
          s.pop();
        } else {
          jack += arr[u];
          other += arr[v];
          status[u] = 0;
          status[v] = 0;
        }
        i = v + 1;
      } else {
        while (v <= n && status[v] != 2)
          v++;
        other += arr[u];
        jack += arr[v];
        status[u] = 0;
        status[v] = 0;
        i = i + 1;
      }
    } else
      i++;
  }
  cout << jack << " " << other << endl;
  return 0;
}