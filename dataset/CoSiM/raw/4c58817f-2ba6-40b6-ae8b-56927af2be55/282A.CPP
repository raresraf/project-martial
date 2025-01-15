#include <bits/stdc++.h>
using namespace std;

int main() {
  int i, n, X = 0;
  cin >> n;
  string op[n];
  for (i = 0; i < n; i++) {
    cin >> op[i];
  }
  for (i = 0; i < n; i++) {
    /*if(i==n-1)
    {
        if(op[i]=="X++")
            cout<<X++<<endl;
        if(op[i]=="++X")
            cout<<++X<<endl;
        if(op[i]=="X--")
            cout<<X--<<endl;
        if(op[i]=="--X")
            cout<<--X<<endl;

        break;
    }*/
    if (op[i] == "X++")
      X++;
    if (op[i] == "++X")
      ++X;
    if (op[i] == "X--")
      X--;
    if (op[i] == "--X")
      --X;
  }
  cout << X << endl;
  return 0;
}
