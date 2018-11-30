import matplotlib.pyplot as pl
import numpy as np

feat_real = ['duration',
             'src_bytes',
             'dst_bytes',
             'wrong_fragment',
             'urgent',
             'hot',
             'num_failed_logins',
             'num_compromied',
             'root_shell',
             'su_attempted',
             'num_root',
             'num_file_creations',
             'num_shells',
]

'duration': real
'src_bytes': real
'dst_bytes': real
'land': {'0', '1'}
'wrong_fragment': real
'urgent': real
'hot': real
'num_failed_logins': real
'logged_in': {'0', '1'}
'num_compromised': real
'root_shell': real
'su_attempted': real
'num_root': real
'num_file_creations': real
'num_shells': real
'num_access_files': real
'num_outbound_cmds': real
'is_host_login': {'0', '1'}
'is_guest_login': {'0', '1'}
'count': real
'srv_count': real
'serror_rate': real
'srv_serror_rate': real
'rerror_rate': real
'srv_rerror_rate': real
'same_srv_rate': real
'diff_srv_rate': real
'srv_diff_host_rate': real
'dst_host_count': real
'dst_host_srv_count': real
'dst_host_same_srv_rate': real
'dst_host_diff_srv_rate': real
'dst_host_same_src_port_rate': real
'dst_host_srv_diff_host_rate': real
'dst_host_serror_rate': real
'dst_host_srv_serror_rate': real
'dst_host_rerror_rate': real
'dst_host_srv_rerror_rate': real
'protocol_type': {'tcp','udp', 'icmp'} 
'service': {'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50'} 
'flag': { 'OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH' }


features = np.genfromtxt('trainData.csv', delimiter=',', dtype=None,
                         encoding=None)


fig, axes = pl.subplots(nrows=7, ncols=7, sharey='row')


for i, name in enumerate(features.dtype.names):
    axes.flat[i].hist(features[name], 20)


pl.show()
