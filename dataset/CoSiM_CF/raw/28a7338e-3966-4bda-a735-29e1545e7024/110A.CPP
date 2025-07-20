/*#include <bits/stdc++.h>
#include<algorithm>
//#include <string.h>
using namespace std;

int main()
{
string s;
long long four,f1,s1,sev,check=0;
cin>>s;
four=count(s.begin(),s.end(),'4');
sev=count(s.begin(),s.end(),'7');
if(four+sev==s.length())
cout<<"YES"<<"\n";
else
{
check=four+sev;
string c=to_string(check);
f1=count(c.begin(),c.end(),'4');
s1=count(c.begin(),c.end(),'7');
if(f1+s1==c.length())
cout<<"YES\n";
else
cout<<"NO\n";
}
return 0;
}
*/
#include <algorithm>
#include <bits/stdc++.h>
//#include <string.h>
using namespace std;

int main() {
  string s;
  long long four, f1, s1, sev, check = 0;
  cin >> s;
  four = count(s.begin(), s.end(), '4');
  sev = count(s.begin(), s.end(), '7');
  // if(four+sev==s.length())
  // cout<<"YES"<<"\n";
  // else
  //{
  check = four + sev;
  string c = to_string(check);
  // f1=count(c.begin(),c.end(),'4');
  // s1=count(c.begin(),c.end(),'7');
  // if(f1+s1==c.length())
  for (int j = 0; j < c.length(); j++)
    if (c[j] != '4' && c[j] != '7') {
      cout << "NO\n";
      break;
    } else
      cout << "YES\n";

  return 0;
}
