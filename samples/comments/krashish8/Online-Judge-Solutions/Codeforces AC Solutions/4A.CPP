/*
Problem Name: A - Watermelon
Problem ID: 4A
Problem URL: https://codeforces.com/contest/4/problem/A
Author: Ashish Kumar (ashishkr23438)
Solution ID: 39498143
Solution Time: 2018-06-21 21:42:22
Language: GNU C++17
Time consumed: 30ms
Memory consumed: 400KB
*/
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

typedef vector<int> vi;
typedef pair<int, int> pi;
typedef long long int ll;

#define speed                                                                  \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define all(c) c.begin(), c.end()
#define allr(c) c.rbegin(), c.rend()
#define present(container, element) (container.find(element) != container.end())
#define cpresent(container, element)                                           \
  (find(all(container), element) != container.end())
#define se second
#define fi first
#define sz(a) int((a).size())
#define mp make_pair
#define pb push_back
#define endl "\n"
#define dbg(x)                                                                 \
  { cout << #x << ": " << (x); }
#define dbg2(x, y)                                                             \
  { cout << #x << ": " << (x) << " , " << #y << ": " << (y) << endl; }

#define INF 1e18;
#define PI 3.14159265359
#define EPS 1e-9
#define MOD 1000000007

const int N = 404040;

bool a[N];

int main() {
  // freopen("input.txt", "r", stdin);
  speed;

  int w;
  cin >> w;
  if (w > 2 && w % 2 == 0)
    cout << "YES";
  else
    cout << "NO";
}