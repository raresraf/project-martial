#include <bits/stdc++.h>
using namespace std;

#define ll long long int
#define ull unsigned long long int
#define pb push_back
#define vpop pop_back
#define mp make_pair
#define ff first
#define ss second
#define clz(x) __builtin_clz(x) // count leading zeroes
#define ctz(x) __builtin_ctz(x) // count trailing zeroes
#define sbts(x) __builtin_popcount(x) // count set bits
#define ps(x, y) fixed << setprecision(y) << x
#define all(v) v.begin(), v.end()
const int MOD = 1e9 + 7;
const ll INF = 1e12;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  int t;
  cin >> t;
  while (t--) {
    int n, arr[20], size = -1, temp;
    cin >> n;
    for (int i = 0; i < n; i++) {
      cin >> temp;
      if (size == -1)
        arr[++size] = temp;
      else if ((arr[size] < 0 && temp < 0) || (arr[size] > 0 && temp > 0))
        arr[size] = max(arr[size], temp);
      else
        arr[++size] = temp;
    }
    ll result = 0;
    for (int i = 0; i <= size; i++)
      result += arr[i];
    cout << result << "\n";
  }
  return 0;
}

// int t;
//     cin >> t;
//     while(t--) {
//         int n, arr[200000], count = 0, pos = -1, neg = 1, pk = 0, pn = 0;
//         cin >> n;
//         for(int i = 0; i < n; i++) {
//             cin >> arr[i];
//             if(arr[i] < 0)  count++;
//             if(pos < 0 && arr[i] > 0)
//                 pos = 1, pk++;
//             else if(pos > 0 && arr[i] < 0)
//                 pos = -1, pk++;
//             if(neg < 0 && arr[i] > 0)
//                 neg = 1, pn++;
//             else if(neg > 0 && arr[i] < 0)
//                 neg = -1, pn++;
//         }
//         int neg_len, pos_len;
//         if(pk < pn)
//             neg_len = (pn + 1) / 2, pos_len = (pn - neg_len);
//         else
//             pos_len = (pk + 1) / 2, neg_len = (pk - pos_len);

//     }