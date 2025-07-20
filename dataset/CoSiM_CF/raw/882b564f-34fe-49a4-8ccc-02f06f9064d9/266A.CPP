#include <iostream>
#include <stack>
using namespace std;
int main() {
  string s;
  int count = 0, num;
  stack<char> st;
  cin >> num;
  cin >> s;
  st.push(s[0]);
  for (int i = 1; i < s.size(); i++) {
    if (st.top() == s[i])
      count++;
    else
      st.push(s[i]);
  }
  cout << count << endl;
  return 0;
}
